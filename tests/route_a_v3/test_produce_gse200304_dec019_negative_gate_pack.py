from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = STAGING_ROOT.parent
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts/route_a_v3/produce_gse200304_dec019_negative_gate_pack.py"
)
CONFIG_PATH = (
    STAGING_ROOT
    / "configs/route_a_v3_gse200304_dec019_negative_gate_pack_v1.json"
)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRODUCER = _load_module(SCRIPT_PATH, "gse200304_dec019_negative_gate_pack")


def _producer_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _producer_i_config() -> dict[str, Any]:
    config = _producer_config()
    binding = config["implementation_binding"]
    binding["status"] = PRODUCER.UNKNOWN
    binding["implementation_commit"] = PRODUCER.UNKNOWN
    binding["implementation_script_sha256"] = PRODUCER.UNKNOWN
    binding["implementation_test_sha256"] = PRODUCER.UNKNOWN
    PRODUCER.validate_static_config(config)
    return config


def _consumer_unbound_i_config(*, base_commit: str | None = None) -> dict[str, Any]:
    config = _producer_i_config()
    if base_commit is not None:
        config["repository_authority"]["base_commit"] = base_commit
        config["repository_authority"]["implementation_commit_expected_parent"] = (
            base_commit
        )
    consumer = config["consumer_authority"]
    consumer["status"] = PRODUCER.UNKNOWN
    consumer["successor_binding_commit"] = PRODUCER.UNKNOWN
    consumer["config_sha256"] = PRODUCER.UNKNOWN
    consumer["script_sha256"] = PRODUCER.UNKNOWN
    consumer["test_sha256"] = PRODUCER.UNKNOWN
    consumer["science_core_sha256"] = PRODUCER.UNKNOWN
    config["implementation_binding"][
        "config_core_sha256"
    ] = PRODUCER.config_core_sha256(config)
    PRODUCER.validate_static_config(config)
    return config


def _producer_bound_consumer_unbound_config() -> dict[str, Any]:
    config = _consumer_unbound_i_config()
    binding = config["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = PRODUCER.sha256(SCRIPT_PATH.read_bytes())
    binding["implementation_test_sha256"] = PRODUCER.sha256(Path(__file__).read_bytes())
    PRODUCER.validate_static_config(config)
    PRODUCER.validate_implementation_binding(config)
    return config


def _consumer_sources() -> dict[str, Path]:
    """Work both in this staging tree and after the three files enter the repo."""

    integrated = {
        "config": STAGING_ROOT
        / "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json",
        "script": STAGING_ROOT
        / "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py",
        "test": STAGING_ROOT
        / "tests/route_a_v3/test_adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py",
    }
    if all(path.is_file() for path in integrated.values()):
        integrated_config = json.loads(integrated["config"].read_text(encoding="utf-8"))
        if (
            integrated_config["implementation_binding"]["status"] == "BOUND"
            and integrated_config["evidence_contract"]["evidence_schema_version"]
            == PRODUCER.EVIDENCE_SCHEMA_VERSION
        ):
            return integrated
    successor_staged = {
        "config": WORK_ROOT
        / "g200_consumer_descriptor_lifecycle_final_staging/configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json",
        "script": WORK_ROOT
        / "g200_consumer_descriptor_lifecycle_final_staging/scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py",
        "test": WORK_ROOT
        / "g200_consumer_descriptor_lifecycle_final_staging/tests/route_a_v3/test_adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py",
    }
    assert all(path.is_file() for path in successor_staged.values())
    return successor_staged


def _consumer_payloads() -> dict[str, bytes]:
    """Return a real B config or an explicit synthetic B from the reviewed I3."""

    sources = _consumer_sources()
    payloads = {key: path.read_bytes() for key, path in sources.items()}
    consumer = json.loads(payloads["config"])
    binding = consumer["implementation_binding"]
    if binding["status"] == PRODUCER.UNKNOWN:
        binding["status"] = "BOUND"
        binding["implementation_commit"] = "a" * 40
        binding["implementation_script_sha256"] = PRODUCER.sha256(payloads["script"])
        binding["implementation_test_sha256"] = PRODUCER.sha256(payloads["test"])
        payloads["config"] = PRODUCER.json_bytes(consumer)
    return payloads


def _predecessor_consumer_config_payload() -> bytes:
    staged = (
        WORK_ROOT
        / "g200_descriptor_d1_staging/configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
    )
    if staged.is_file():
        payload = staged.read_bytes()
        assert PRODUCER.sha256(payload) == PRODUCER.PREDECESSOR_DESCRIPTOR_CONFIG_SHA256
        return payload

    history = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(STAGING_ROOT),
            "rev-list",
            "--all",
            "--",
            PRODUCER.CONSUMER_CONFIG_REPO_PATH,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    for commit in history:
        result = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(STAGING_ROOT),
                "show",
                f"{commit}:{PRODUCER.CONSUMER_CONFIG_REPO_PATH}",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if (
            result.returncode == 0
            and PRODUCER.sha256(result.stdout)
            == PRODUCER.PREDECESSOR_DESCRIPTOR_CONFIG_SHA256
        ):
            return result.stdout
    raise AssertionError("frozen predecessor consumer config fixture was not found")


def _consumer_rebound_i_config(
    *,
    current_descriptor_base_commit: str | None = None,
    successor_binding_commit: str | None = None,
) -> dict[str, Any]:
    current = _producer_i_config()
    if (
        current["consumer_authority"]["status"] == "BOUND"
        and current_descriptor_base_commit is None
        and successor_binding_commit is None
    ):
        PRODUCER.validate_consumer_authority_binding(current)
        return current
    current_descriptor_base_commit = current_descriptor_base_commit or "2" * 40
    successor_binding_commit = successor_binding_commit or "3" * 40
    prior_base = (
        current["repository_authority"]["base_commit"]
        if current["repository_authority"]["base_commit"]
        != current_descriptor_base_commit
        else "8" * 40
    )
    before = _consumer_unbound_i_config(base_commit=prior_base)
    config = copy.deepcopy(before)
    payloads = _consumer_payloads()
    repository = config["repository_authority"]
    repository["base_commit"] = current_descriptor_base_commit
    repository["implementation_commit_expected_parent"] = (
        current_descriptor_base_commit
    )
    consumer = config["consumer_authority"]
    consumer["status"] = "BOUND"
    consumer["successor_binding_commit"] = successor_binding_commit
    consumer["config_sha256"] = PRODUCER.sha256(payloads["config"])
    consumer["script_sha256"] = PRODUCER.sha256(payloads["script"])
    consumer["test_sha256"] = PRODUCER.sha256(payloads["test"])
    source_consumer = json.loads(payloads["config"])
    consumer["science_core_sha256"] = source_consumer["implementation_binding"][
        "config_core_sha256"
    ]
    config["implementation_binding"][
        "config_core_sha256"
    ] = PRODUCER.config_core_sha256(config)
    PRODUCER.validate_pre_i_consumer_rebase(
        before,
        config,
        current_descriptor_base_commit=current_descriptor_base_commit,
        successor_binding_commit=successor_binding_commit,
        consumer_config_sha256=consumer["config_sha256"],
        consumer_script_sha256=consumer["script_sha256"],
        consumer_test_sha256=consumer["test_sha256"],
        consumer_science_core_sha256=consumer["science_core_sha256"],
    )
    return config


def _bound_config() -> dict[str, Any]:
    config = _consumer_rebound_i_config()
    binding = config["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = PRODUCER.sha256(SCRIPT_PATH.read_bytes())
    binding["implementation_test_sha256"] = PRODUCER.sha256(Path(__file__).read_bytes())
    PRODUCER.validate_static_config(config)
    PRODUCER.validate_implementation_binding(config)
    PRODUCER.validate_consumer_authority_binding(config)
    return config


def _materialize_consumer_repo(root: Path, config: dict[str, Any]) -> None:
    payloads = _consumer_payloads()
    authority = config["consumer_authority"]
    for source_key, path_key in {
        "config": "config_path",
        "script": "script_path",
        "test": "test_path",
    }.items():
        destination = root / authority[path_key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[source_key])


def _consumer_context(
    tmp_path: Path,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], ModuleType, Path]:
    bound = config or _bound_config()
    repo = tmp_path / "repo"
    _materialize_consumer_repo(repo, bound)
    consumer, module = PRODUCER._load_verified_consumer(bound, repo=repo)
    return bound, consumer, module, repo


def _records(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], ModuleType, dict[str, bytes], Path]:
    config, consumer, module, repo = _consumer_context(tmp_path)
    payloads = PRODUCER.build_records(config, consumer, module)
    return config, consumer, module, payloads, repo


def _record_by_gate(payloads: dict[str, bytes], gate_id: str) -> dict[str, Any]:
    for payload in payloads.values():
        record = json.loads(payload)
        if record["gate_id"] == gate_id:
            return record
    raise AssertionError(f"missing gate: {gate_id}")


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        assert not (
            {key.casefold() for key in value}
            & PRODUCER.FORBIDDEN_OUTPUT_KEY_TOKENS
        )
        for child in value.values():
            _assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_keys(child)


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _build_synthetic_consumer_lifecycle(
    repo: Path,
    *,
    claimed_implementation_commit: str | None = None,
    extra_i_path: bool = False,
    extra_b_scalar: bool = False,
) -> dict[str, Any]:
    consumer_config_path = repo / PRODUCER.CONSUMER_CONFIG_REPO_PATH
    consumer_config_path.parent.mkdir(parents=True, exist_ok=True)
    consumer_config_path.write_bytes(_predecessor_consumer_config_payload())
    seed = repo / "AUTHORITY_BASE.txt"
    seed.write_text("synthetic authority base\n", encoding="utf-8")
    _git(repo, "add", "AUTHORITY_BASE.txt", PRODUCER.CONSUMER_CONFIG_REPO_PATH)
    _git(repo, "commit", "-q", "-m", "synthetic predecessor descriptor")
    consumer_base_commit = _git(repo, "rev-parse", "HEAD")

    consumer_sources = _consumer_sources()
    script_payload = consumer_sources["script"].read_bytes()
    test_payload = consumer_sources["test"].read_bytes()
    consumer_module = _load_module(
        consumer_sources["script"],
        f"synthetic_consumer_lifecycle_{repo.name}",
    )
    i_config = json.loads(consumer_sources["config"].read_text(encoding="utf-8"))
    repository = i_config["repository_authority"]
    repository["base_commit"] = consumer_base_commit
    repository["implementation_commit_expected_parent"] = consumer_base_commit
    i_binding = i_config["implementation_binding"]
    i_binding["status"] = PRODUCER.UNKNOWN
    i_binding["implementation_commit"] = PRODUCER.UNKNOWN
    i_binding["implementation_script_sha256"] = PRODUCER.UNKNOWN
    i_binding["implementation_test_sha256"] = PRODUCER.UNKNOWN
    i_binding["config_core_sha256"] = consumer_module.config_core_sha256(i_config)
    consumer_config_path.write_bytes(PRODUCER.json_bytes(i_config))
    for relative, payload in {
        PRODUCER.CONSUMER_SCRIPT_REPO_PATH: script_payload,
        PRODUCER.CONSUMER_TEST_REPO_PATH: test_payload,
    }.items():
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    i_paths = list(PRODUCER.CONSUMER_IMPLEMENTATION_PATHS)
    if extra_i_path:
        extra = repo / "UNAUTHORIZED_CONSUMER_I_PATH.txt"
        extra.write_text("not part of consumer I\n", encoding="utf-8")
        i_paths.append(extra.name)
    _git(repo, "add", *i_paths)
    _git(repo, "commit", "-q", "-m", "synthetic consumer I")
    implementation_commit = _git(repo, "rev-parse", "HEAD")

    b_config = copy.deepcopy(i_config)
    b_binding = b_config["implementation_binding"]
    b_binding["status"] = "BOUND"
    b_binding["implementation_commit"] = (
        claimed_implementation_commit or implementation_commit
    )
    b_binding["implementation_script_sha256"] = PRODUCER.sha256(script_payload)
    b_binding["implementation_test_sha256"] = PRODUCER.sha256(test_payload)
    if extra_b_scalar:
        b_config["current_external_state"]["qualified"] = True
    consumer_config_path.write_bytes(PRODUCER.json_bytes(b_config))
    _git(repo, "add", PRODUCER.CONSUMER_CONFIG_REPO_PATH)
    _git(repo, "commit", "-q", "-m", "synthetic consumer B")
    binding_commit = _git(repo, "rev-parse", "HEAD")
    return {
        "base_commit": consumer_base_commit,
        "implementation_commit": implementation_commit,
        "binding_commit": binding_commit,
        "config": b_config,
        "payloads": {
            "config": consumer_config_path.read_bytes(),
            "script": script_payload,
            "test": test_payload,
        },
    }


def test_unknown_config_is_exact_and_stops_before_source_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _consumer_unbound_i_config()
    PRODUCER.validate_static_config(config)
    binding = config["implementation_binding"]
    assert binding["status"] == PRODUCER.UNKNOWN
    assert binding["implementation_commit"] == PRODUCER.UNKNOWN
    assert binding["implementation_script_sha256"] == PRODUCER.UNKNOWN
    assert binding["implementation_test_sha256"] == PRODUCER.UNKNOWN
    consumer = config["consumer_authority"]
    assert consumer["status"] == PRODUCER.UNKNOWN
    assert consumer["successor_binding_commit"] == PRODUCER.UNKNOWN
    assert consumer["config_sha256"] == PRODUCER.UNKNOWN
    assert consumer["script_sha256"] == PRODUCER.UNKNOWN
    assert consumer["test_sha256"] == PRODUCER.UNKNOWN
    assert consumer["science_core_sha256"] == PRODUCER.UNKNOWN
    assert PRODUCER.config_core_sha256(config) == binding["config_core_sha256"]

    calls = {"source": 0, "output": 0, "git": 0}

    def source_forbidden(*_args: Any, **_kwargs: Any) -> None:
        calls["source"] += 1
        raise AssertionError("consumer source opened before binding")

    def output_forbidden(*_args: Any, **_kwargs: Any) -> None:
        calls["output"] += 1
        raise AssertionError("output touched before binding")

    def git_forbidden(*_args: Any, **_kwargs: Any) -> None:
        calls["git"] += 1
        raise AssertionError("git authority inspected before binding")

    monkeypatch.setattr(PRODUCER, "_load_verified_consumer", source_forbidden)
    monkeypatch.setattr(PRODUCER, "publish_records", output_forbidden)
    monkeypatch.setattr(PRODUCER, "validate_production_authority", git_forbidden)
    with pytest.raises(PRODUCER.BindingError, match="BINDING_UNKNOWN_NOT_ASSERTED"):
        PRODUCER.produce(
            config,
            tmp_path / "must-not-exist",
            production=True,
            repo=tmp_path / "must-not-open",
        )
    assert calls == {"source": 0, "output": 0, "git": 0}
    assert not (tmp_path / "must-not-exist").exists()


def test_consumer_unknown_stops_after_producer_binding_but_before_source_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _producer_bound_consumer_unbound_config()
    calls = {"source": 0, "output": 0}

    def source_forbidden(*_args: Any, **_kwargs: Any) -> None:
        calls["source"] += 1
        raise AssertionError("consumer source opened before successor authority binding")

    def output_forbidden(*_args: Any, **_kwargs: Any) -> None:
        calls["output"] += 1
        raise AssertionError("output touched before successor authority binding")

    monkeypatch.setattr(PRODUCER, "_load_verified_consumer", source_forbidden)
    monkeypatch.setattr(PRODUCER, "publish_records", output_forbidden)
    with pytest.raises(
        PRODUCER.BindingError,
        match="FINAL_CONSUMER_SUCCESSOR_AUTHORITY_UNKNOWN_NOT_ASSERTED",
    ):
        PRODUCER.produce(
            config,
            tmp_path / "must-not-exist",
            production=False,
            repo=tmp_path / "must-not-open",
        )
    assert calls == {"source": 0, "output": 0}
    assert not (tmp_path / "must-not-exist").exists()


def test_pre_i_consumer_and_current_descriptor_base_rebind_is_closed() -> None:
    after = _consumer_rebound_i_config(
        current_descriptor_base_commit="4" * 40,
        successor_binding_commit="5" * 40,
    )
    current_base = _producer_config()["repository_authority"]["base_commit"]
    before = _consumer_unbound_i_config(
        base_commit=current_base if current_base != "4" * 40 else "8" * 40
    )
    assert PRODUCER._scalar_differences(before, after) == (
        PRODUCER.PRE_I_REBASE_SCALAR_PATHS
    )
    assert PRODUCER.PRE_I_REBASE_SCALAR_PATHS == {
        "implementation_binding.config_core_sha256",
        "repository_authority.base_commit",
        "repository_authority.implementation_commit_expected_parent",
        "consumer_authority.status",
        "consumer_authority.successor_binding_commit",
        "consumer_authority.config_sha256",
        "consumer_authority.script_sha256",
        "consumer_authority.test_sha256",
        "consumer_authority.science_core_sha256",
    }
    assert after["repository_authority"]["base_commit"] == "4" * 40
    assert after["consumer_authority"]["successor_binding_commit"] == "5" * 40
    assert after["implementation_binding"]["status"] == PRODUCER.UNKNOWN
    PRODUCER.validate_static_config(after)
    PRODUCER.validate_consumer_authority_binding(after)

    tampered = copy.deepcopy(after)
    tampered["negative_gate_records"][0]["status"] = "PASS"
    tampered["implementation_binding"][
        "config_core_sha256"
    ] = PRODUCER.config_core_sha256(tampered)
    with pytest.raises(PRODUCER.ProducerError):
        PRODUCER.validate_pre_i_consumer_rebase(
            before,
            tampered,
            current_descriptor_base_commit="4" * 40,
            successor_binding_commit="5" * 40,
            consumer_config_sha256=after["consumer_authority"]["config_sha256"],
            consumer_script_sha256=after["consumer_authority"]["script_sha256"],
            consumer_test_sha256=after["consumer_authority"]["test_sha256"],
            consumer_science_core_sha256=after["consumer_authority"][
                "science_core_sha256"
            ],
        )


def test_consumer_successor_preserves_science_descriptor_and_v3_identity() -> None:
    source = _consumer_sources()["config"]
    predecessor = json.loads(source.read_text(encoding="utf-8"))
    successor = copy.deepcopy(predecessor)
    successor["repository_authority"]["base_commit"] = "6" * 40
    successor["repository_authority"][
        "implementation_commit_expected_parent"
    ] = "6" * 40
    successor["implementation_binding"]["implementation_commit"] = "7" * 40
    successor["implementation_binding"]["config_core_sha256"] = "9" * 64
    authority = _consumer_rebound_i_config()["consumer_authority"]

    PRODUCER.validate_consumer_science_continuity(
        predecessor,
        successor,
        authority,
    )
    assert (
        PRODUCER.consumer_non_repository_science_projection_sha256(successor)
        == PRODUCER.REQUIRED_NON_REPOSITORY_SCIENCE_PROJECTION_SHA256
    )
    assert (
        successor["evidence_descriptor_bindings"]["descriptor_set_sha256"]
        == PRODUCER.REQUIRED_DESCRIPTOR_SET_SHA256
    )
    assert PRODUCER._consumer_v3_identity(successor)["evidence_schema_version"] == (
        PRODUCER.EVIDENCE_SCHEMA_VERSION
    )
    assert PRODUCER._consumer_v3_identity(successor)["acceptance_rule"] == (
        PRODUCER.ACCEPTANCE_RULE
    )

    tampered = copy.deepcopy(successor)
    tampered["evidence_contract"]["gate_record_provenance_contract"][
        "acceptance_authority"
    ]["rule"] = "CONFIG_HASH_BOUND_ACCEPTED_AGGREGATE_GATE_RECORD_V2"
    with pytest.raises(PRODUCER.ProducerError, match="science projection"):
        PRODUCER.validate_consumer_science_continuity(
            predecessor,
            tampered,
            authority,
        )


def test_exact_consumer_production_authority_result_must_match_pack_claims() -> None:
    config = _consumer_rebound_i_config()
    authority = config["consumer_authority"]
    producer_git_authority = {
        "consumer_successor_base_commit": "4" * 40,
        "consumer_successor_implementation_commit": "5" * 40,
        "current_head": "6" * 40,
    }
    expected = {
        "lifecycle_state": "REPAIR_B_BOUND_OR_DESCRIPTOR_DESCENDANT",
        "repair_base_commit": producer_git_authority[
            "consumer_successor_base_commit"
        ],
        "repair_implementation_commit": producer_git_authority[
            "consumer_successor_implementation_commit"
        ],
        "repair_binding_commit": authority["successor_binding_commit"],
        "current_head": producer_git_authority["current_head"],
        "science_core_sha256": authority["science_core_sha256"],
        "evidence_descriptor_set_sha256": authority[
            "required_descriptor_set_sha256"
        ],
    }
    module = ModuleType("synthetic_verified_consumer_authority")
    module.validate_production_authority = lambda _consumer: copy.deepcopy(expected)
    assert (
        PRODUCER.validate_verified_consumer_production_authority(
            config,
            {},
            module,
            producer_git_authority,
        )
        == expected
    )

    wrong = copy.deepcopy(expected)
    wrong["repair_binding_commit"] = "7" * 40
    module.validate_production_authority = lambda _consumer: wrong
    with pytest.raises(PRODUCER.ProducerError, match="repair_binding_commit"):
        PRODUCER.validate_verified_consumer_production_authority(
            config,
            {},
            module,
            producer_git_authority,
        )


def test_i_to_b_transition_is_exactly_four_binding_scalars() -> None:
    initial = _consumer_rebound_i_config()
    bound = _bound_config()
    PRODUCER.validate_i_to_b_transition(
        initial,
        bound,
        implementation_commit="1" * 40,
        implementation_script_sha256=bound["implementation_binding"][
            "implementation_script_sha256"
        ],
        implementation_test_sha256=bound["implementation_binding"][
            "implementation_test_sha256"
        ],
    )

    tampered = copy.deepcopy(bound)
    tampered["record_policy"]["training_allowed"] = True
    with pytest.raises(PRODUCER.ProducerError):
        PRODUCER.validate_i_to_b_transition(
            initial,
            tampered,
            implementation_commit="1" * 40,
            implementation_script_sha256=bound["implementation_binding"][
                "implementation_script_sha256"
            ],
            implementation_test_sha256=bound["implementation_binding"][
                "implementation_test_sha256"
            ],
        )


def test_real_git_i_b_clean_pushed_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", PRODUCER.BRANCH, os.fspath(repo)],
        check=True,
    )
    _git(repo, "config", "user.name", "Route A Test")
    _git(repo, "config", "user.email", "route-a-test@example.invalid")

    before_rebase = _consumer_unbound_i_config()
    before_rebase["repository_authority"]["production_repo_root"] = os.fspath(repo)
    before_rebase["implementation_binding"][
        "config_core_sha256"
    ] = PRODUCER.config_core_sha256(before_rebase)
    monkeypatch.setattr(PRODUCER, "PRODUCTION_REPO_ROOT", repo)
    consumer_lifecycle = _build_synthetic_consumer_lifecycle(repo)
    consumer_config_path = repo / PRODUCER.CONSUMER_CONFIG_REPO_PATH
    consumer_base_commit = consumer_lifecycle["base_commit"]
    consumer_implementation_commit = consumer_lifecycle["implementation_commit"]
    consumer_binding_commit = consumer_lifecycle["binding_commit"]
    consumer_payloads = consumer_lifecycle["payloads"]

    initial = copy.deepcopy(before_rebase)
    initial["repository_authority"]["base_commit"] = consumer_binding_commit
    initial["repository_authority"]["implementation_commit_expected_parent"] = (
        consumer_binding_commit
    )
    consumer = initial["consumer_authority"]
    consumer["status"] = "BOUND"
    consumer["successor_binding_commit"] = consumer_binding_commit
    consumer["config_sha256"] = PRODUCER.sha256(consumer_config_path.read_bytes())
    consumer["script_sha256"] = PRODUCER.sha256(consumer_payloads["script"])
    consumer["test_sha256"] = PRODUCER.sha256(consumer_payloads["test"])
    consumer_config = json.loads(consumer_config_path.read_text(encoding="utf-8"))
    consumer["science_core_sha256"] = consumer_config["implementation_binding"][
        "config_core_sha256"
    ]
    initial["implementation_binding"][
        "config_core_sha256"
    ] = PRODUCER.config_core_sha256(initial)
    PRODUCER.validate_pre_i_consumer_rebase(
        before_rebase,
        initial,
        current_descriptor_base_commit=consumer_binding_commit,
        successor_binding_commit=consumer_binding_commit,
        consumer_config_sha256=consumer["config_sha256"],
        consumer_script_sha256=consumer["script_sha256"],
        consumer_test_sha256=consumer["test_sha256"],
        consumer_science_core_sha256=consumer["science_core_sha256"],
    )
    for source, relative in {
        CONFIG_PATH: PRODUCER.CONFIG_REPO_PATH,
        SCRIPT_PATH: PRODUCER.SCRIPT_REPO_PATH,
        Path(__file__): PRODUCER.TEST_REPO_PATH,
    }.items():
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source == CONFIG_PATH:
            destination.write_bytes(PRODUCER.json_bytes(initial))
        else:
            shutil.copyfile(source, destination)
    _git(repo, "add", *PRODUCER.IMPLEMENTATION_PATHS)
    _git(repo, "commit", "-q", "-m", "producer I")
    implementation = _git(repo, "rev-parse", "HEAD")

    bound = copy.deepcopy(initial)
    binding = bound["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = implementation
    binding["implementation_script_sha256"] = PRODUCER.sha256(
        (repo / PRODUCER.SCRIPT_REPO_PATH).read_bytes()
    )
    binding["implementation_test_sha256"] = PRODUCER.sha256(
        (repo / PRODUCER.TEST_REPO_PATH).read_bytes()
    )
    (repo / PRODUCER.CONFIG_REPO_PATH).write_bytes(PRODUCER.json_bytes(bound))
    _git(repo, "add", PRODUCER.CONFIG_REPO_PATH)
    _git(repo, "commit", "-q", "-m", "producer B")
    binding_commit = _git(repo, "rev-parse", "HEAD")

    subprocess.run(["git", "init", "-q", "--bare", os.fspath(remote)], check=True)
    _git(repo, "remote", "add", "origin", os.fspath(remote))
    _git(repo, "push", "-q", "-u", "origin", PRODUCER.BRANCH)
    authority = PRODUCER.validate_production_authority(bound, repo=repo)
    assert authority == {
        "base_commit": consumer_binding_commit,
        "consumer_successor_binding_commit": consumer_binding_commit,
        "consumer_successor_implementation_commit": consumer_implementation_commit,
        "consumer_successor_base_commit": consumer_base_commit,
        "implementation_commit": implementation,
        "binding_commit": binding_commit,
        "current_head": binding_commit,
        "upstream_head": binding_commit,
    }

    external_hardlink = tmp_path / "producer-script-hardlink"
    os.link(repo / PRODUCER.SCRIPT_REPO_PATH, external_hardlink)
    with pytest.raises(PRODUCER.ScopeViolation, match="single-link regular"):
        PRODUCER.validate_production_authority(bound, repo=repo)
    external_hardlink.unlink()

    (repo / PRODUCER.SCRIPT_REPO_PATH).write_text("tamper\n", encoding="utf-8")
    with pytest.raises(PRODUCER.AuthorityError, match="not clean"):
        PRODUCER.validate_production_authority(bound, repo=repo)


def test_git_tree_symlink_is_not_a_regular_bound_producer_blob(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", os.fspath(repo)], check=True)
    _git(repo, "config", "user.name", "Route A Test")
    _git(repo, "config", "user.email", "route-a-test@example.invalid")
    target = repo / "target.py"
    target.write_text("print('target')\n", encoding="utf-8")
    linked = repo / "linked.py"
    linked.symlink_to(target.name)
    _git(repo, "add", "target.py", "linked.py")
    _git(repo, "commit", "-q", "-m", "symlink fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(PRODUCER.AuthorityError, match="regular Git blob"):
        PRODUCER._git_regular_blob(
            repo,
            commit,
            "linked.py",
            label="linked producer",
        )


@pytest.mark.parametrize(
    ("variant", "options", "message"),
    [
        (
            "missing_consumer_i",
            {"claimed_implementation_commit": "f" * 40},
            "implementation commit",
        ),
        ("extra_consumer_i_path", {"extra_i_path": True}, "exact three-file"),
        ("extra_consumer_b_scalar", {"extra_b_scalar": True}, "exact four-scalar"),
    ],
)
def test_consumer_successor_lifecycle_rejects_false_b_authority(
    tmp_path: Path,
    variant: str,
    options: dict[str, Any],
    message: str,
) -> None:
    repo = tmp_path / variant
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", PRODUCER.BRANCH, os.fspath(repo)],
        check=True,
    )
    _git(repo, "config", "user.name", "Route A Test")
    _git(repo, "config", "user.email", "route-a-test@example.invalid")
    lifecycle = _build_synthetic_consumer_lifecycle(repo, **options)
    config = lifecycle["config"]
    binding = config["implementation_binding"]
    authority = {
        "script_sha256": PRODUCER.sha256(lifecycle["payloads"]["script"]),
        "test_sha256": PRODUCER.sha256(lifecycle["payloads"]["test"]),
        "science_core_sha256": binding["config_core_sha256"],
    }
    with pytest.raises(PRODUCER.ProducerError, match=message):
        PRODUCER.validate_consumer_successor_lifecycle(
            repo=repo,
            successor_binding_commit=lifecycle["binding_commit"],
            current_head=lifecycle["binding_commit"],
            current_consumer_config=config,
            authority=authority,
        )


def test_seven_records_match_closed_status_reason_fact_and_privacy_schema(
    tmp_path: Path,
) -> None:
    config, consumer, module, payloads, _repo = _records(tmp_path)
    assert tuple(sorted(payloads)) == PRODUCER.MEMBER_NAMES
    assert len(payloads) == 7
    predecessor = consumer["evidence_contract"]["required_predecessor_authority"]
    acceptance = consumer["evidence_contract"]["gate_record_provenance_contract"][
        "acceptance_authority"
    ]
    assert PRODUCER.EVIDENCE_SCHEMA_VERSION.endswith(".v3")
    assert PRODUCER.EVIDENCE_RECORD_TYPE.endswith("_V3")
    assert acceptance["rule"] == PRODUCER.ACCEPTANCE_RULE
    slots = {slot["slot_id"]: slot for slot in consumer["evidence_contract"]["slots"]}

    for spec in PRODUCER.GATE_SPECS:
        payload = payloads[spec["allowed_basename"]]
        record = json.loads(payload)
        assert set(record) == module.COMMON_EVIDENCE_KEYS
        assert record["schema_version"] == module.EVIDENCE_SCHEMA_VERSION
        assert record["record_type"] == module.EVIDENCE_RECORD_TYPE
        assert record["gate_id"] == spec["gate_id"]
        assert record["status"] == spec["status"]
        assert record["accepted"] is True
        assert record["aggregate_only"] is True
        assert record["facts"] is None
        assert record["unknown_fields"] == sorted(module.FACT_KEYS[spec["gate_id"]])
        assert record["reason_codes"] == spec["reason_codes"]
        assert record["reason_codes"] == sorted(set(record["reason_codes"]))
        assert record["privacy"] == PRODUCER.PRIVACY
        assert record["provenance"]["predecessor_authority"] == predecessor
        assert record["provenance"]["acceptance_authority"] == acceptance
        assert (
            record["provenance"]["source_bundle_root_or_target_sha256"]
            == PRODUCER.SOURCE_TARGET_SHA256
        )
        assert "member_hashes" not in record["provenance"]
        _assert_no_forbidden_keys(record)
        assert (
            module._validate_gate_record(
                payload,
                slots[spec["gate_id"]],
                consumer,
            )
            == record
        )

    assert config["record_policy"]["ordinary_study_contribution_delta"] == 0
    assert config["record_policy"]["a1_study_contribution_delta"] == 0
    assert config["record_policy"]["true_a2_study_contribution_delta"] == 0
    assert config["record_policy"]["canonical_record_count_delta"] == 0
    assert config["record_policy"]["training_allowed"] is False
    assert config["record_policy"]["model_selection_allowed"] is False
    assert config["record_policy"]["next_phase_authorized"] is False


def test_statuses_and_reasons_are_the_exact_authorized_seven(tmp_path: Path) -> None:
    _config, _consumer, _module, payloads, _repo = _records(tmp_path)
    observed = {
        record["gate_id"]: (record["status"], record["reason_codes"])
        for record in (json.loads(payload) for payload in payloads.values())
    }
    assert observed == {
        "CANONICAL_REPORTED_ENDPOINT_SEMANTICS": (
            "BLOCKED",
            ["AUTHOR_COUNTING_POLICY_AND_PAPER_FAITHFUL_MAPPING_NOT_CLOSED"],
        ),
        "BIOLOGICAL_GROUP_AUTHORITY": (
            "BLOCKED",
            ["BIOLOGICAL_GROUP_AUTHORITY_NOT_CLOSED"],
        ),
        "ROW_REPLICATE_OR_VALID_SE": (
            "BLOCKED",
            ["ROW_LEVEL_REPLICATE_OR_VALID_SE_NOT_ESTABLISHED"],
        ),
        "CHECKPOINT_SPECIFIC_EXPOSURE": (
            "UNKNOWN_NOT_ASSERTED",
            ["CHECKPOINT_SPECIFIC_EXPOSURE_UNKNOWN_NOT_ASSERTED"],
        ),
        "LICENSE_RIGHTS": (
            "UNKNOWN_NOT_ASSERTED",
            ["LICENSE_RIGHTS_UNKNOWN_NOT_ASSERTED"],
        ),
        "OUTCOME_BLIND_SPLIT_LEAKAGE": (
            "NOT_RUN",
            ["OUTCOME_BLIND_SPLIT_LEAKAGE_NOT_RUN"],
        ),
        "PREFROZEN_POWER_PRECISION": (
            "NOT_RUN",
            ["PREFROZEN_POWER_PRECISION_NOT_RUN"],
        ),
    }
    assert "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE" not in observed
    encoded = b"\n".join(payloads.values())
    assert b"80S" not in encoded
    assert b"RAW_REPLAY_INDEPENDENT_REPRODUCTION" not in encoded


def test_producer_rejects_false_pass_and_consumer_rejects_zero_and_tamper(
    tmp_path: Path,
) -> None:
    _config, consumer, module, payloads, _repo = _records(tmp_path)
    slots = {slot["slot_id"]: slot for slot in consumer["evidence_contract"]["slots"]}
    gate_id = "PREFROZEN_POWER_PRECISION"
    original = _record_by_gate(payloads, gate_id)

    false_pass_config = _bound_config()
    power_spec = next(
        spec
        for spec in false_pass_config["negative_gate_records"]
        if spec["gate_id"] == gate_id
    )
    power_spec["status"] = "PASS"
    false_pass_config["implementation_binding"][
        "config_core_sha256"
    ] = PRODUCER.config_core_sha256(false_pass_config)
    with pytest.raises(PRODUCER.ProducerError, match="negative gate specs"):
        PRODUCER.validate_static_config(false_pass_config)
    assert original["status"] == "NOT_RUN" and original["facts"] is None

    numeric_zero = copy.deepcopy(original)
    numeric_zero["facts"] = {key: 0 for key in original["unknown_fields"]}
    with pytest.raises(module.AdjudicationError, match="facts=null"):
        module._validate_gate_record(
            PRODUCER.json_bytes(numeric_zero), slots[gate_id], consumer
        )

    provenance_tamper = copy.deepcopy(original)
    provenance_tamper["provenance"][
        "source_bundle_root_or_target_sha256"
    ] = "0" * 64
    with pytest.raises(module.AdjudicationError, match="source target identity"):
        module._validate_gate_record(
            PRODUCER.json_bytes(provenance_tamper), slots[gate_id], consumer
        )


def test_verified_consumer_config_tamper_is_rejected(tmp_path: Path) -> None:
    config = _bound_config()
    repo = tmp_path / "repo"
    _materialize_consumer_repo(repo, config)
    path = repo / config["consumer_authority"]["config_path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["current_external_state"]["qualified"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PRODUCER.AuthorityError, match="config_path SHA differs"):
        PRODUCER._load_verified_consumer(config, repo=repo)


def test_publish_is_atomic_exact_and_idempotent(tmp_path: Path) -> None:
    config, _consumer, _module, payloads, _repo = _records(tmp_path)
    output = tmp_path / "final-pack"
    assert (
        PRODUCER.publish_records(
            output,
            payloads,
            production=False,
            config=config,
        )
        == "PUBLISHED"
    )
    assert sorted(path.name for path in output.iterdir()) == sorted(payloads)
    for name, expected in payloads.items():
        path = output / name
        assert path.is_file()
        assert path.stat().st_nlink == 1
        assert path.read_bytes() == expected
    assert (
        PRODUCER.publish_records(
            output,
            payloads,
            production=False,
            config=config,
        )
        == "EXISTING_EXACT"
    )


def test_publisher_fault_before_rename_leaves_no_final_directory(
    tmp_path: Path,
) -> None:
    config, _consumer, _module, payloads, _repo = _records(tmp_path)
    output = tmp_path / "final-pack"

    def fault(phase: str) -> None:
        if phase.startswith("after_write:"):
            raise RuntimeError("injected pre-rename failure")

    with pytest.raises(RuntimeError, match="injected pre-rename failure"):
        PRODUCER.publish_records(
            output,
            payloads,
            production=False,
            config=config,
            fault_injector=fault,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".final-pack.tmp.*"))


def test_publisher_refuses_parent_replacement_before_atomic_rename(
    tmp_path: Path,
) -> None:
    config, _consumer, _module, payloads, _repo = _records(tmp_path)
    parent = tmp_path / "publication-root"
    parent.mkdir()
    output = parent / "final-pack"
    moved_parent = tmp_path / "moved-publication-root"

    def fault(phase: str) -> None:
        if phase == "before_atomic_rename":
            parent.rename(moved_parent)
            parent.mkdir()

    with pytest.raises(PRODUCER.PartialPublicationError, match="temp preserved"):
        PRODUCER.publish_records(
            output,
            payloads,
            production=False,
            config=config,
            fault_injector=fault,
        )
    assert not output.exists()
    assert not list(parent.iterdir())
    assert list(moved_parent.glob(".final-pack.tmp.*"))


def test_precommit_cleanup_never_deletes_same_name_temp_replacement(
    tmp_path: Path,
) -> None:
    config, _consumer, _module, payloads, _repo = _records(tmp_path)
    parent = tmp_path / "publication-root"
    parent.mkdir()
    output = parent / "final-pack"
    moved_temp = tmp_path / "moved-original-temp"
    replacement_member = sorted(payloads)[0]
    replacement_path: Path | None = None

    def fault(phase: str) -> None:
        nonlocal replacement_path
        if phase.startswith("after_write:"):
            temp = next(parent.glob(".final-pack.tmp.*"))
            temp.rename(moved_temp)
            replacement = parent / temp.name
            replacement.mkdir()
            replacement_path = replacement / replacement_member
            replacement_path.write_bytes(b"replacement-must-survive")
            raise RuntimeError("replace unpublished temp")

    with pytest.raises(PRODUCER.PartialPublicationError, match="temp preserved"):
        PRODUCER.publish_records(
            output,
            payloads,
            production=False,
            config=config,
            fault_injector=fault,
        )
    assert replacement_path is not None
    assert replacement_path.read_bytes() == b"replacement-must-survive"
    assert moved_temp.is_dir()
    assert not output.exists()


def test_existing_exact_never_uses_a_renamed_parent_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _consumer, _module, payloads, _repo = _records(tmp_path)
    parent = tmp_path / "publication-root"
    parent.mkdir()
    output = parent / "final-pack"
    moved_parent = tmp_path / "moved-publication-root"
    assert PRODUCER.publish_records(
        output,
        payloads,
        production=False,
        config=config,
    ) == "PUBLISHED"

    def parent_replace_then_exists(
        _parent_fd: int,
        _old_name: str,
        _new_name: str,
    ) -> None:
        parent.rename(moved_parent)
        parent.mkdir()
        raise FileExistsError("injected existing target")

    monkeypatch.setattr(
        PRODUCER,
        "_native_rename_noreplace",
        parent_replace_then_exists,
    )
    with pytest.raises(PRODUCER.PartialPublicationError, match="cleanup failed"):
        PRODUCER.publish_records(
            output,
            payloads,
            production=False,
            config=config,
        )
    assert not output.exists()
    assert (moved_parent / "final-pack").is_dir()


def test_post_rename_fault_reports_committed_exact_state(tmp_path: Path) -> None:
    config, _consumer, _module, payloads, _repo = _records(tmp_path)
    output = tmp_path / "final-pack"

    def fault(phase: str) -> None:
        if phase == "after_atomic_rename":
            raise RuntimeError("injected post-rename failure")

    with pytest.raises(PRODUCER.PublicationStateError) as captured:
        PRODUCER.publish_records(
            output,
            payloads,
            production=False,
            config=config,
            fault_injector=fault,
        )
    assert captured.value.publication_state == "COMMITTED_EXACT"
    assert output.is_dir()
    assert (
        PRODUCER.publish_records(
            output,
            payloads,
            production=False,
            config=config,
        )
        == "EXISTING_EXACT"
    )


def test_post_rename_parent_replacement_is_never_reported_committed_exact(
    tmp_path: Path,
) -> None:
    config, _consumer, _module, payloads, _repo = _records(tmp_path)
    parent = tmp_path / "publication-root"
    parent.mkdir()
    output = parent / "final-pack"
    moved_parent = tmp_path / "moved-publication-root"

    def fault(phase: str) -> None:
        if phase == "after_atomic_rename":
            parent.rename(moved_parent)
            parent.mkdir()
            raise RuntimeError("replace parent after commit")

    with pytest.raises(PRODUCER.PublicationStateError) as captured:
        PRODUCER.publish_records(
            output,
            payloads,
            production=False,
            config=config,
            fault_injector=fault,
        )
    assert captured.value.publication_state == "COMMITTED_UNVERIFIED"
    assert not output.exists()
    assert (moved_parent / "final-pack").is_dir()


def test_existing_partial_or_different_target_is_never_overwritten(
    tmp_path: Path,
) -> None:
    config, _consumer, _module, payloads, _repo = _records(tmp_path)
    output = tmp_path / "final-pack"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_bytes(b"preserve-me")
    with pytest.raises(PRODUCER.PartialPublicationError, match="partial|extra"):
        PRODUCER.publish_records(
            output,
            payloads,
            production=False,
            config=config,
        )
    assert sentinel.read_bytes() == b"preserve-me"
    assert sorted(path.name for path in output.iterdir()) == ["sentinel.txt"]


def test_symlink_parent_and_hardlinked_member_are_rejected(tmp_path: Path) -> None:
    config, _consumer, _module, payloads, _repo = _records(tmp_path)
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    symlink_parent = tmp_path / "alias"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(PRODUCER.ScopeViolation, match="symlink"):
        PRODUCER.publish_records(
            symlink_parent / "final-pack",
            payloads,
            production=False,
            config=config,
        )

    output = tmp_path / "hardlink-pack"
    output.mkdir()
    names = sorted(payloads)
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    extra_link = tmp_path / "extra-link"
    os.link(output / names[0], extra_link)
    with pytest.raises(PRODUCER.ScopeViolation, match="single-link"):
        PRODUCER.publish_records(
            output,
            payloads,
            production=False,
            config=config,
        )
    assert extra_link.read_bytes() == payloads[names[0]]


def test_produce_summary_does_not_change_any_gate_count(tmp_path: Path) -> None:
    config = _bound_config()
    repo = tmp_path / "repo"
    _materialize_consumer_repo(repo, config)
    output = tmp_path / "final-pack"
    result = PRODUCER.produce(
        config,
        output,
        production=False,
        repo=repo,
    )
    assert result["record_count"] == 7
    assert result["all_records_consumer_accepted"] is True
    assert result["ordinary_study_contribution_delta"] == 0
    assert result["a1_study_contribution_delta"] == 0
    assert result["true_a2_study_contribution_delta"] == 0
    assert result["canonical_record_count_delta"] == 0
    assert result["training_allowed"] is False
    assert result["model_selection_allowed"] is False
    assert result["next_phase_authorized"] is False
