"""CPU-only regression tests for strict MK0 GPU runtime provenance."""

from __future__ import annotations

import importlib.util
import json
import os
from argparse import Namespace
from pathlib import Path
import py_compile
import subprocess
import sys
import threading
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_gpu_runner() -> Any:
    path = ROOT / "scripts" / "mk0" / "run_mk0_gpu_smoke.py"
    spec = importlib.util.spec_from_file_location("mk0_gpu_runtime_binding_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_finalizer() -> Any:
    path = ROOT / "scripts" / "mk0" / "finalize_mk0_acceptance.py"
    spec = importlib.util.spec_from_file_location("mk0_finalizer_binding_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding() -> dict[str, Any]:
    return {
        "goal_sha256": "a" * 64,
        "implementation_commit": "b" * 40,
        "source_binding": {"tracked_source_files_sha256": "c" * 64},
    }


def _test_generator_only() -> None:
    return None


def _test_rate_interface() -> None:
    return None


def _test_generator_and_rate() -> None:
    _test_rate_interface()


_EXTERNAL_CALLBACK: Any = None


def _test_rate_with_external() -> None:
    assert _EXTERNAL_CALLBACK is not None
    _EXTERNAL_CALLBACK()


def _test_generator_with_external() -> None:
    _test_rate_with_external()


def _test_generator_with_thread() -> None:
    _test_rate_interface()
    assert _EXTERNAL_CALLBACK is not None
    thread = threading.Thread(target=_EXTERNAL_CALLBACK, name="mk0-role-audit-test")
    thread.start()
    thread.join()


def _external_callback(
    tmp_path: Path,
    *,
    module_name: str,
    function_name: str,
) -> Any:
    namespace: dict[str, Any] = {"__name__": module_name}
    filename = tmp_path / f"{module_name.replace('.', '_')}_{function_name}.py"
    exec(
        compile(
            f"def {function_name}():\n    return None\n",
            str(filename),
            "exec",
            dont_inherit=True,
            optimize=0,
        ),
        namespace,
    )
    return namespace[function_name]


def _external_recorder(runner: Any, *, run_id: str) -> Any:
    records, _production_metadata = runner._formal_gpu_role_interface_records()
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    exact_metadata = {
        _test_generator_with_external.__code__: (
            relative,
            _test_generator_with_external.__qualname__,
            _test_generator_with_external.__code__.co_firstlineno,
            ("generator_interface",),
        ),
        _test_generator_with_thread.__code__: (
            relative,
            _test_generator_with_thread.__qualname__,
            _test_generator_with_thread.__code__.co_firstlineno,
            ("generator_interface",),
        ),
        _test_rate_with_external.__code__: (
            relative,
            _test_rate_with_external.__qualname__,
            _test_rate_with_external.__code__.co_firstlineno,
            ("rate_interface",),
        ),
        _test_rate_interface.__code__: (
            relative,
            _test_rate_interface.__qualname__,
            _test_rate_interface.__code__.co_firstlineno,
            ("rate_interface",),
        ),
    }
    return runner._FormalGpuRoleQueryRecorder(
        _binding(),
        run_id=run_id,
        interface_records=records,
        role_code_metadata=exact_metadata,
    )


def planned_gpu_rates() -> None:
    """Deliberately rate-like name that must not receive a role category."""

    return None


def test_strict_bootstrap_purges_stale_descendant_and_editable_finder(
    tmp_path: Path,
) -> None:
    stale_root = tmp_path / "stale_install" / "mrna_editflow"
    stale_module = stale_root / "core" / "mk0" / "foundation_fusion.py"
    stale_module.parent.mkdir(parents=True)
    for initializer in (
        stale_root / "__init__.py",
        stale_root / "core" / "__init__.py",
        stale_root / "core" / "mk0" / "__init__.py",
    ):
        initializer.write_text("", encoding="utf-8")
    stale_module.write_text("ORIGIN = 'stale'\n", encoding="utf-8")

    program = r"""
import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
stale_parent = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(stale_parent))
stale = importlib.import_module("mrna_editflow.core.mk0.foundation_fusion")
assert getattr(stale, "ORIGIN") == "stale"

class EditableFinder(importlib.abc.MetaPathFinder):
    triggered = False
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("mrna_editflow"):
            self.triggered = True
            raise AssertionError("editable finder must be disabled")
        return None

finder = EditableFinder()
sys.meta_path.insert(0, finder)
class PoisonLoader(importlib.abc.Loader):
    def create_module(self, spec): return None
    def exec_module(self, module):
        module.__file__ = str(root / "core" / "__init__.py")
        module.__path__ = [str(root / "core")]
        module.POISONED = True
class PoisonPathFinder(importlib.abc.PathEntryFinder):
    triggered = False
    def find_spec(self, fullname, target=None):
        if fullname == "mrna_editflow.core":
            self.triggered = True
            return importlib.util.spec_from_loader(
                fullname, PoisonLoader(), is_package=True
            )
        return None
path_finder = PoisonPathFinder()
sys.path_importer_cache[str(root)] = path_finder
helper_path = root / "scripts" / "mk0" / "strict_worktree_import.py"
spec = importlib.util.spec_from_file_location("strict_bootstrap_test", helper_path)
helper = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(helper)
with helper.strict_worktree_package_import(root):
    pass
current = importlib.import_module("mrna_editflow.core.mk0.foundation_fusion")
assert Path(current.__file__).resolve() == root / "core" / "mk0" / "foundation_fusion.py"
assert current is not stale
assert not getattr(sys.modules["mrna_editflow.core"], "POISONED", False)
assert not finder.triggered
assert not path_finder.triggered
assert finder in sys.meta_path
assert getattr(sys.meta_path[0], "_mk0_strict_finder_marker", None)
assert current.__cached__ is None
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(ROOT), str(stale_root.parent)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_strict_bootstrap_ignores_timestamp_valid_poisoned_pyc(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "poison_pkg"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    probe = package_root / "probe.py"
    poisoned = "VALUE = 'poison'\n"
    source = "VALUE = 'source'\n"
    assert len(poisoned) == len(source)
    timestamp = 1_700_000_000
    probe.write_text(poisoned, encoding="utf-8")
    os.utime(probe, (timestamp, timestamp))
    pyc = Path(py_compile.compile(str(probe), doraise=True))
    probe.write_text(source, encoding="utf-8")
    os.utime(probe, (timestamp, timestamp))
    assert pyc.is_file()

    program = r"""
import importlib
import importlib.util
from pathlib import Path
import sys

package_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(package_root.parent))
baseline = importlib.import_module("poison_pkg.probe")
assert baseline.VALUE == "poison"
helper_path = Path(sys.argv[2]).resolve()
source = helper_path.read_bytes()
helper = type(sys)("strict_bootstrap_source_test")
helper.__file__ = str(helper_path)
helper.__cached__ = None
exec(compile(source, str(helper_path), "exec", dont_inherit=True, optimize=0), helper.__dict__)
with helper.strict_worktree_package_import(package_root, "poison_pkg"):
    current = importlib.import_module("poison_pkg.probe")
assert current.VALUE == "source"
assert current.__cached__ is None
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(package_root),
            str(ROOT / "scripts" / "mk0" / "strict_worktree_import.py"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_role_categories_use_exact_production_code_identities() -> None:
    runner = _load_gpu_runner()
    records, role_metadata = runner._formal_gpu_role_interface_records()
    by_label = {record["interface"]: record for record in records}

    assert by_label["foundation_fusion.rate_field_forward"]["role_categories"] == [
        "rate_interface"
    ]
    assert by_label["foundation_fusion.official_paper_adapter"]["role_categories"] == [
        "rate_interface"
    ]
    assert role_metadata[runner.FoundationFusionRateField.forward.__code__][3] == (
        "rate_interface",
    )

    recorder = runner._FormalGpuRoleQueryRecorder(
        _binding(),
        run_id="unit-role-identities",
        interface_records=records,
        role_code_metadata=role_metadata,
    )
    constructor_metadata = recorder._metadata_for_code(
        runner.FoundationFusionRateField.__init__.__code__
    )
    named_helper_metadata = recorder._metadata_for_code(planned_gpu_rates.__code__)
    assert constructor_metadata is not None
    assert "rate_interface" not in constructor_metadata[3]
    assert named_helper_metadata is not None
    assert "rate_interface" not in named_helper_metadata[3]
    assert runner._FormalGpuRoleQueryRecorder._prohibited_categories(
        "core/mk0/critic_runtime.py", "forward"
    ) == ("critic_query",)
    assert runner._FormalGpuRoleQueryRecorder._prohibited_categories(
        "core/mk0/final_evaluator_runtime.py", "forward"
    ) == ("final_evaluator_query",)


def test_missing_exact_rate_call_preserves_partial_phase_evidence(
    tmp_path: Path,
) -> None:
    runner = _load_gpu_runner()
    records, _production_metadata = runner._formal_gpu_role_interface_records()
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    exact_metadata = {
        _test_generator_only.__code__: (
            relative,
            _test_generator_only.__qualname__,
            _test_generator_only.__code__.co_firstlineno,
            ("generator_interface",),
        )
    }
    recorder = runner._FormalGpuRoleQueryRecorder(
        _binding(),
        run_id="unit-partial-evidence",
        interface_records=records,
        role_code_metadata=exact_metadata,
    )

    with pytest.raises(runner.SmokeFailure, match="lacks rate_interface") as caught:
        recorder.run_phase("generator_rate_official_frozen_arm", _test_generator_only)

    evidence = caught.value.partial_phase_evidence
    assert evidence["status"] == "FAILED_WITH_PARTIAL_EVIDENCE"
    assert evidence["formal_gpu_computation_complete"] is False
    assert evidence["formal_gpu_phase_count"] == 1
    assert evidence["completed_phase_count"] == 1
    assert evidence["failed_phase_count"] == 1
    phase = evidence["phase_records"][0]
    assert phase["phase_status"] == "FAILED"
    assert phase["generator_interface_call_count"] == 1
    assert phase["rate_interface_call_count"] == 0
    assert phase["failure_reason"] == str(caught.value)

    partial_model_loading_policy = {
        "schema_version": "mk0_model_loading_thread_policy_v1",
        "status": "PASS",
    }
    caught.value.partial_model_loading_policy = partial_model_loading_policy

    failure_path = runner._write_failure_best_effort(
        tmp_path,
        run_id="unit-partial-evidence",
        snapshot_dir="/not-loaded",
        device="cuda:0",
        error=caught.value,
    )
    assert failure_path is not None
    persisted = json.loads(failure_path.read_text(encoding="utf-8"))
    assert persisted["partial_phase_evidence"] == evidence
    assert persisted["partial_model_loading_policy"] == partial_model_loading_policy


def test_exact_generator_and_rate_code_calls_satisfy_phase() -> None:
    runner = _load_gpu_runner()
    records, _production_metadata = runner._formal_gpu_role_interface_records()
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    exact_metadata = {
        _test_generator_and_rate.__code__: (
            relative,
            _test_generator_and_rate.__qualname__,
            _test_generator_and_rate.__code__.co_firstlineno,
            ("generator_interface",),
        ),
        _test_rate_interface.__code__: (
            relative,
            _test_rate_interface.__qualname__,
            _test_rate_interface.__code__.co_firstlineno,
            ("rate_interface",),
        ),
    }
    recorder = runner._FormalGpuRoleQueryRecorder(
        _binding(),
        run_id="unit-exact-positive",
        interface_records=records,
        role_code_metadata=exact_metadata,
    )

    recorder.run_phase("generator_rate_official_frozen_arm", _test_generator_and_rate)
    phase = recorder.partial_evidence()["phase_records"][0]
    assert phase["phase_status"] == "PASS"
    assert phase["generator_interface_call_count"] == 1
    assert phase["rate_interface_call_count"] == 1


def test_external_critic_call_fails_closed_with_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_gpu_runner()
    recorder = _external_recorder(runner, run_id="unit-external-critic")
    callback = _external_callback(
        tmp_path,
        module_name="thirdparty.critic_sdk",
        function_name="score_candidate",
    )
    monkeypatch.setattr(sys.modules[__name__], "_EXTERNAL_CALLBACK", callback)

    with pytest.raises(runner.SmokeFailure, match="prohibited role query") as caught:
        recorder.run_phase(
            "generator_rate_official_frozen_arm", _test_generator_with_external
        )

    phase = caught.value.partial_phase_evidence["phase_records"][0]
    critic = [
        record
        for record in phase["external_call_inventory"]
        if "critic_query" in record["categories"]
    ]
    assert critic
    assert all(record["classification"] == "prohibited_role" for record in critic)
    assert phase["critic_query_call_count"] > 0


def test_unknown_external_call_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_gpu_runner()
    recorder = _external_recorder(runner, run_id="unit-external-unknown")
    callback = _external_callback(
        tmp_path,
        module_name="thirdparty.benign_sdk",
        function_name="calculate",
    )
    monkeypatch.setattr(sys.modules[__name__], "_EXTERNAL_CALLBACK", callback)

    with pytest.raises(runner.SmokeFailure, match="unknown external") as caught:
        recorder.run_phase(
            "generator_rate_official_frozen_arm", _test_generator_with_external
        )

    phase = caught.value.partial_phase_evidence["phase_records"][0]
    assert phase["unknown_external_call_count"] > 0
    assert any(
        record["module_name"] == "thirdparty.benign_sdk"
        and record["classification"] == "unknown_external"
        for record in phase["external_call_inventory"]
    )


def test_repository_generated_pseudo_source_is_classified_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_gpu_runner()
    recorder = _external_recorder(runner, run_id="unit-repository-generated")
    namespace: dict[str, Any] = {"__name__": "mrna_editflow.core.mk0.types"}
    exec(
        compile(
            "def generated_repository_callable():\n    return None\n",
            "<string>",
            "exec",
            dont_inherit=True,
            optimize=0,
        ),
        namespace,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_EXTERNAL_CALLBACK",
        namespace["generated_repository_callable"],
    )

    recorder.run_phase(
        "generator_rate_official_frozen_arm", _test_generator_with_external
    )
    phase = recorder.partial_evidence()["phase_records"][0]
    assert phase["phase_status"] == "PASS"
    assert phase["unknown_external_call_count"] == 0
    assert any(
        record["module_name"] == "mrna_editflow.core.mk0.types"
        and record["source_file"] == "<string>"
        and record["classification"] == runner.REPOSITORY_GENERATED_CLASSIFICATION
        for record in phase["external_call_inventory"]
    )
    finalizer = _load_finalizer()
    assert finalizer._external_call_classification(
        "mrna_editflow.core.mk0.types", "<string>", "__init__", 2
    ) == (finalizer.REPOSITORY_GENERATED_CLASSIFICATION, ())


def test_frozen_foundation_prefix_remains_viable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_gpu_runner()
    recorder = _external_recorder(runner, run_id="unit-external-allowed")
    callback = _external_callback(
        tmp_path,
        module_name="torch.synthetic_test",
        function_name="calculate",
    )
    monkeypatch.setattr(sys.modules[__name__], "_EXTERNAL_CALLBACK", callback)

    recorder.run_phase(
        "generator_rate_official_frozen_arm", _test_generator_with_external
    )
    phase = recorder.partial_evidence()["phase_records"][0]
    assert phase["phase_status"] == "PASS"
    assert phase["unknown_external_call_count"] == 0
    assert any(
        record["module_name"] == "torch.synthetic_test"
        and record["classification"] == "frozen_foundation_stack_allowlist"
        for record in phase["external_call_inventory"]
    )


def test_typing_extensions_runtime_dependency_is_allowlisted_exactly(
    tmp_path: Path,
) -> None:
    runner = _load_gpu_runner()
    binding, code_identities = runner._typing_compatibility_shim_state()
    assert binding["distribution_version"] == "4.16.0"
    assert binding["source_sha256"] == runner.TYPING_COMPATIBILITY_SHIM_SOURCE_SHA256
    assert binding["module_match"] == "exact_only_no_submodules"
    assert binding["requirements_lock"]["reproduced"] is False
    assert len(binding["allowed_calls"]) == 4
    for record in binding["allowed_calls"]:
        key = (record["function_qualname"], record["first_lineno"])
        assert runner._external_call_classification(
            "typing_extensions",
            binding["source_file"],
            key[0],
            key[1],
            code_identities[key],
        ) == (runner.TYPING_COMPATIBILITY_SHIM_CLASSIFICATION, ())

    fake = _external_callback(
        tmp_path,
        module_name="typing_extensions",
        function_name="_collect_type_vars",
    )
    assert runner._external_call_classification(
        "typing_extensions",
        str(Path(fake.__code__.co_filename).resolve()),
        fake.__qualname__,
        fake.__code__.co_firstlineno,
        fake.__code__,
    ) == ("unknown_external", ())
    assert runner._external_call_classification(
        "typing_extensions.unrelated_sdk",
        binding["source_file"],
        "calculate",
        1,
        None,
    ) == ("unknown_external", ())


def test_typing_extensions_allowlist_does_not_hide_prohibited_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_gpu_runner()
    recorder = _external_recorder(runner, run_id="unit-typing-extensions-prohibited")
    callback = _external_callback(
        tmp_path,
        module_name="typing_extensions.critic_sdk",
        function_name="reward_candidate",
    )
    monkeypatch.setattr(sys.modules[__name__], "_EXTERNAL_CALLBACK", callback)

    with pytest.raises(runner.SmokeFailure, match="prohibited role query") as caught:
        recorder.run_phase(
            "generator_rate_official_frozen_arm", _test_generator_with_external
        )

    phase = caught.value.partial_phase_evidence["phase_records"][0]
    assert any(
        record["module_name"] == "typing_extensions.critic_sdk"
        and record["classification"] == "prohibited_role"
        and "critic_query" in record["categories"]
        and "final_evaluator_query" in record["categories"]
        for record in phase["external_call_inventory"]
    )


def test_external_critic_in_new_thread_fails_closed_with_thread_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_gpu_runner()
    recorder = _external_recorder(runner, run_id="unit-thread-critic")
    callback = _external_callback(
        tmp_path,
        module_name="thirdparty.worker",
        function_name="critic_reward",
    )
    monkeypatch.setattr(sys.modules[__name__], "_EXTERNAL_CALLBACK", callback)

    with pytest.raises(runner.SmokeFailure, match="prohibited role query") as caught:
        recorder.run_phase(
            "generator_rate_official_frozen_arm", _test_generator_with_thread
        )

    phase = caught.value.partial_phase_evidence["phase_records"][0]
    assert phase["python_thread_count"] >= 2
    assert any(
        record["thread_name"] == "mk0-role-audit-test"
        and record["external_call_count"] > 0
        for record in phase["thread_inventory"]
    )
    assert phase["critic_query_call_count"] > 0


def test_preexisting_noncurrent_thread_fails_before_formal_callback() -> None:
    runner = _load_gpu_runner()
    records, _production_metadata = runner._formal_gpu_role_interface_records()
    recorder = runner._FormalGpuRoleQueryRecorder(
        _binding(),
        run_id="unit-preexisting-thread",
        interface_records=records,
        role_code_metadata={},
    )
    ready = threading.Event()
    release = threading.Event()
    callback_executed = False

    def background() -> None:
        ready.set()
        release.wait(timeout=10)

    def formal_callback() -> None:
        nonlocal callback_executed
        callback_executed = True

    thread = threading.Thread(target=background, name="preexisting-critic")
    thread.start()
    assert ready.wait(timeout=5)
    try:
        with pytest.raises(
            runner.SmokeFailure, match="preexisting noncurrent Python threads"
        ) as caught:
            recorder.run_phase("generator_rate_official_frozen_arm", formal_callback)
        phase = caught.value.partial_phase_evidence["phase_records"][0]
        assert callback_executed is False
        assert phase["completed"] is False
        assert phase["preexisting_noncurrent_python_thread_count"] == 1
        assert phase["preexisting_noncurrent_python_threads"] == [
            {
                "thread_id": thread.ident,
                "native_id": thread.native_id,
                "thread_name": "preexisting-critic",
                "daemon": False,
            }
        ]
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.mark.parametrize("mode", ["success", "body_exception"])
def test_scoped_model_loading_policy_controls_real_tqdm_apis_in_clean_process(
    mode: str,
) -> None:
    script = r"""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

runner_path = Path(sys.argv[1])
mode = sys.argv[2]
spec = importlib.util.spec_from_file_location("mk0_progress_subprocess", runner_path)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from huggingface_hub.utils import tqdm as hub_tqdm
from tqdm.auto import tqdm as auto_tqdm
from tqdm.std import tqdm as standard_tqdm
from transformers.utils import logging as transformers_logging

classes = (standard_tqdm, auto_tqdm, hub_tqdm)
hub_progress_module = importlib.import_module("huggingface_hub.utils.tqdm")
hub_progress_registry = hub_progress_module.progress_bar_states
assert len({id(cls) for cls in classes}) == 3
if mode == "success":
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    transformers_logging.enable_progress_bar()
else:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    transformers_logging.disable_progress_bar()
hub_progress_registry.clear()
hub_progress_registry.update({"_global": True, "probe.named": False})
hub_registry_before = dict(hub_progress_registry)
hub_registry_identity_before = id(hub_progress_registry)
environment_before = (
    "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ,
    os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"),
)
progress_before = transformers_logging.is_progress_bar_enabled()
class_state_before = [
    ("monitor_interval" in cls.__dict__, getattr(cls, "monitor_interval"))
    for cls in classes
]

class Marker(RuntimeError):
    pass

policy = None
try:
    with runner._scoped_model_loading_thread_policy() as active_policy:
        policy = active_policy
        assert all(
            "monitor_interval" in cls.__dict__
            and type(cls.__dict__["monitor_interval"]) is int
            and cls.__dict__["monitor_interval"] == 0
            for cls in classes
        )
        for api in (
            transformers_logging.tqdm,
            standard_tqdm,
            auto_tqdm,
            hub_tqdm,
        ):
            progress = api(range(2), disable=True)
            list(progress)
            close = getattr(progress, "close", None)
            if close is not None:
                close()
        assert runner._live_noncurrent_python_threads() == []
        runner._mark_model_load_completed(active_policy, "official_frozen")
        runner._mark_model_load_completed(
            active_policy, "same_architecture_from_scratch_control"
        )
        if mode == "body_exception":
            raise Marker("intentional body exception")
except Marker as error:
    assert mode == "body_exception"
    policy = error.partial_model_loading_policy
else:
    assert mode == "success"

assert policy is not None
assert runner._live_noncurrent_python_threads() == []
assert [
    ("monitor_interval" in cls.__dict__, getattr(cls, "monitor_interval"))
    for cls in classes
] == class_state_before
assert transformers_logging.is_progress_bar_enabled() is progress_before
assert hub_progress_module.progress_bar_states is hub_progress_registry
assert id(hub_progress_module.progress_bar_states) == hub_registry_identity_before
assert dict(hub_progress_registry) == hub_registry_before
assert (
    "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ,
    os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"),
) == environment_before
assert policy["all_monitor_states_restored"] is True
assert policy["transformers_progress_state_restored"] is True
assert policy["hf_hub_env_state_restored"] is True
assert policy["huggingface_hub_progress_registry_before"] == hub_registry_before
assert policy["huggingface_hub_progress_registry_during_load"] == {"_global": False}
assert policy["huggingface_hub_progress_registry_after_restore"] == hub_registry_before
assert policy["huggingface_hub_progress_registry_restored"] is True
assert policy["huggingface_hub_progress_registry_object_identity_preserved"] is True
assert policy["thread_name_allowlist"] == []
assert policy["thread_cleanup_attempted"] is False
print(json.dumps(policy, sort_keys=True))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "scripts" / "mk0" / "run_mk0_gpu_smoke.py"),
            mode,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    policy = json.loads(completed.stdout.splitlines()[-1])
    assert policy["status"] == ("PASS" if mode == "success" else "FAILED")


def test_model_loading_policy_rejects_any_preexisting_thread_without_cleanup() -> None:
    runner = _load_gpu_runner()
    ready = threading.Event()
    release = threading.Event()
    body_entered = False

    def background() -> None:
        ready.set()
        release.wait(timeout=10)

    thread = threading.Thread(target=background, name="Thread-1", daemon=True)
    thread.start()
    assert ready.wait(timeout=5)
    try:
        with pytest.raises(
            runner.SmokeFailure, match="preexisting noncurrent Python threads"
        ) as caught:
            with runner._scoped_model_loading_thread_policy():
                body_entered = True
        policy = caught.value.partial_model_loading_policy
        assert body_entered is False
        assert policy["noncurrent_python_threads_before_model_loading"] == [
            {
                "thread_id": thread.ident,
                "native_id": thread.native_id,
                "thread_name": "Thread-1",
                "daemon": True,
            }
        ]
        assert policy["thread_name_allowlist"] == []
        assert policy["thread_name_filtering_used"] is False
        assert policy["thread_cleanup_attempted"] is False
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_formal_foundation_pair_covers_both_loads_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_gpu_runner()
    from huggingface_hub.utils import tqdm as hub_tqdm
    from tqdm.auto import tqdm as auto_tqdm
    from tqdm.std import tqdm as standard_tqdm

    classes = (standard_tqdm, auto_tqdm, hub_tqdm)
    before = [
        ("monitor_interval" in cls.__dict__, getattr(cls, "monitor_interval"))
        for cls in classes
    ]
    observed_from_scratch = []

    def fake_load(*_args: Any, from_scratch: bool, **_kwargs: Any) -> tuple[Any, Any]:
        observed_from_scratch.append(from_scratch)
        assert all(
            "monitor_interval" in cls.__dict__
            and type(cls.__dict__["monitor_interval"]) is int
            and cls.__dict__["monitor_interval"] == 0
            for cls in classes
        )
        assert runner._live_noncurrent_python_threads() == []
        return object(), object()

    monkeypatch.setattr(runner, "load_official_utrlm", fake_load)
    result = runner._load_formal_foundation_pair("/unit/snapshot", object())
    policy = result[-1]
    assert observed_from_scratch == [False, True]
    assert policy["status"] == "PASS"
    assert policy["load_completion"] == {
        "official_frozen": True,
        "same_architecture_from_scratch_control": True,
    }
    assert [
        ("monitor_interval" in cls.__dict__, getattr(cls, "monitor_interval"))
        for cls in classes
    ] == before
    later_error = runner.SmokeFailure("later formal phase failed")
    runner._attach_active_partial_model_loading_policy(later_error)
    assert later_error.partial_model_loading_policy is policy


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_model_load_failure_preserves_restored_partial_policy(
    monkeypatch: pytest.MonkeyPatch, fail_on_call: int
) -> None:
    runner = _load_gpu_runner()
    call_count = 0

    def fake_load(*_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == fail_on_call:
            raise RuntimeError(f"load-{fail_on_call}-failed")
        return object(), object()

    monkeypatch.setattr(runner, "load_official_utrlm", fake_load)
    with pytest.raises(RuntimeError, match=f"load-{fail_on_call}-failed") as caught:
        runner._load_formal_foundation_pair("/unit/snapshot", object())
    policy = caught.value.partial_model_loading_policy
    assert policy["status"] == "FAILED"
    assert policy["all_monitor_states_restored"] is True
    assert policy["transformers_progress_state_restored"] is True
    assert policy["hf_hub_env_state_restored"] is True
    assert policy["noncurrent_python_threads_after_policy_restoration"] == []
    assert policy["load_completion"]["official_frozen"] is (fail_on_call == 2)
    assert policy["load_completion"]["same_architecture_from_scratch_control"] is False


def test_thread_starting_after_preexisting_snapshot_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_gpu_runner()
    records, _production_metadata = runner._formal_gpu_role_interface_records()
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    recorder = runner._FormalGpuRoleQueryRecorder(
        _binding(),
        run_id="unit-between-thread-scans",
        interface_records=records,
        role_code_metadata={
            _test_generator_and_rate.__code__: (
                relative,
                _test_generator_and_rate.__qualname__,
                _test_generator_and_rate.__code__.co_firstlineno,
                ("generator_interface",),
            ),
            _test_rate_interface.__code__: (
                relative,
                _test_rate_interface.__qualname__,
                _test_rate_interface.__code__.co_firstlineno,
                ("rate_interface",),
            ),
        },
    )
    ready = threading.Event()
    release = threading.Event()

    def between_snapshots() -> None:
        ready.set()
        release.wait(timeout=10)

    thread = threading.Thread(target=between_snapshots, name="between-scans")
    original_enumerate = threading.enumerate
    enumerate_call_count = 0

    def enumerate_with_race() -> list[threading.Thread]:
        nonlocal enumerate_call_count
        enumerate_call_count += 1
        snapshot = original_enumerate()
        if enumerate_call_count == 1:
            thread.start()
            assert ready.wait(timeout=5)
        return snapshot

    monkeypatch.setattr(threading, "enumerate", enumerate_with_race)
    try:
        with pytest.raises(
            runner.SmokeFailure, match="left new Python threads running"
        ) as caught:
            recorder.run_phase(
                "generator_rate_official_frozen_arm", _test_generator_and_rate
            )
        phase = caught.value.partial_phase_evidence["phase_records"][0]
        assert phase["completed"] is True
        assert phase["preexisting_noncurrent_python_thread_count"] == 0
        assert phase["unjoined_new_thread_ids"] == [thread.ident]
        assert phase["phase_status"] == "FAILED"
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_runtime_interface_origin_failure_precedes_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_gpu_runner()
    loaded = False

    monkeypatch.setattr(runner, "_validate_formal_source_binding", lambda **_kwargs: {})

    def reject_origin(_binding: Any) -> None:
        raise runner.SmokeFailure("runtime interface origin mismatch")

    def mark_model_load(*_args: Any, **_kwargs: Any) -> None:
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(
        runner, "_validate_formal_runtime_interface_origins", reject_origin
    )
    monkeypatch.setattr(runner, "load_official_utrlm", mark_model_load)

    with pytest.raises(runner.SmokeFailure, match="runtime interface origin mismatch"):
        runner.run_gpu_smoke(
            output_dir=tmp_path,
            run_id="unit-origin-order",
            snapshot_dir=tmp_path,
            device_text="cuda:0",
            goal_sha256="a" * 64,
            implementation_commit="b" * 40,
            run_manifest_path=tmp_path / "run_manifest.json",
            preflight_path=tmp_path / "preflight.json",
        )
    assert loaded is False


def test_runtime_interface_origins_match_bound_source_hashes() -> None:
    runner = _load_gpu_runner()
    records, _metadata = runner._formal_gpu_role_interface_records()
    tracked = {
        record["source_file"]: record["source_file_sha256"] for record in records
    }
    binding = {"source_binding": {"tracked_source_files": tracked}}

    validated_records, validated_metadata = (
        runner._validate_formal_runtime_interface_origins(binding)
    )
    assert validated_records == records
    assert len(validated_metadata) == len(records)

    tracked["core/mk0/foundation_fusion.py"] = "0" * 64
    with pytest.raises(runner.SmokeFailure, match="runtime interface hash differs"):
        runner._validate_formal_runtime_interface_origins(binding)


@pytest.mark.parametrize(
    ("error", "expected_reason"),
    [(KeyboardInterrupt(), "KeyboardInterrupt"), (RuntimeError(), "RuntimeError")],
)
def test_gpu_main_writes_nonempty_standard_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: BaseException,
    expected_reason: str,
) -> None:
    runner = _load_gpu_runner()
    run_root = tmp_path / "gpu_failure"
    output_dir = run_root / "artifacts" / "mk0"
    output_dir.mkdir(parents=True)
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({"run_id": "MK0_FAILURE_REASON_TEST"}), encoding="utf-8"
    )
    args = Namespace(
        output_dir=output_dir,
        run_id="MK0_FAILURE_REASON_TEST",
        goal_sha256="a" * 64,
        implementation_commit="b" * 40,
        run_manifest=manifest_path,
        preflight_record=tmp_path / "preflight.json",
        snapshot_dir=tmp_path / "snapshot",
        device="cuda:0",
    )
    observed: dict[str, Any] = {}
    partial_model_loading_policy = {
        "schema_version": "mk0_model_loading_thread_policy_v1",
        "status": "PASS",
    }
    error.partial_model_loading_policy = partial_model_loading_policy

    monkeypatch.setattr(runner, "_parse_args", lambda _argv: args)
    monkeypatch.setattr(runner, "_candidate_bound_run_root", lambda _args: run_root)
    monkeypatch.setattr(
        runner, "resume_failure_closure_if_present", lambda *_args, **_kwargs: None
    )

    def raise_failure(_args: Any, _argv: list[str]) -> Any:
        raise error

    monkeypatch.setattr(runner, "_validate_gpu_launch_contract", raise_failure)
    monkeypatch.setattr(
        runner, "_write_failure_best_effort", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "write_failed_sentinel",
        lambda _root, **kwargs: observed.update(kwargs),
    )

    assert runner.main([]) == 1
    assert observed["reason"] == expected_reason
    persisted = json.loads(
        (run_root / "failure" / "gpu_smoke_failure.json").read_text(encoding="utf-8")
    )
    assert persisted["exception_message"] == expected_reason
    assert persisted["partial_model_loading_policy"] == partial_model_loading_policy


def test_gpu_main_stdout_failure_preserves_active_loading_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_gpu_runner()
    run_root = tmp_path / "gpu_stdout_failure"
    output_dir = run_root / "artifacts" / "mk0"
    output_dir.mkdir(parents=True)
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({"run_id": "MK0_STDOUT_FAILURE_TEST"}), encoding="utf-8"
    )
    args = Namespace(
        output_dir=output_dir,
        run_id="MK0_STDOUT_FAILURE_TEST",
        goal_sha256="a" * 64,
        implementation_commit="b" * 40,
        run_manifest=manifest_path,
        preflight_record=tmp_path / "preflight.json",
        snapshot_dir=tmp_path / "snapshot",
        device="cuda:0",
    )
    partial_policy = {
        "schema_version": "mk0_model_loading_thread_policy_v1",
        "status": "PASS",
    }

    monkeypatch.setattr(runner, "_parse_args", lambda _argv: args)
    monkeypatch.setattr(runner, "_candidate_bound_run_root", lambda _args: run_root)
    monkeypatch.setattr(
        runner, "resume_failure_closure_if_present", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "_validate_gpu_launch_contract",
        lambda _args, _argv: (run_root, {}),
    )

    def successful_gpu_smoke(**_kwargs: Any) -> dict[str, Any]:
        runner._ACTIVE_MODEL_LOADING_POLICY = partial_policy
        return {"device_uuid": "GPU-unit", "max_memory_allocated_bytes": 1}

    monkeypatch.setattr(runner, "run_gpu_smoke", successful_gpu_smoke)
    monkeypatch.setattr(runner, "append_jsonl", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "append_text", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "update_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner, "_write_failure_best_effort", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(runner, "write_failed_sentinel", lambda *_args, **_kwargs: None)
    print_calls = 0

    def fail_first_print(*_args: Any, **_kwargs: Any) -> None:
        nonlocal print_calls
        print_calls += 1
        if print_calls == 1:
            raise BrokenPipeError("unit stdout closed")

    monkeypatch.setattr("builtins.print", fail_first_print)

    assert runner.main([]) == 1
    persisted = json.loads(
        (run_root / "failure" / "gpu_smoke_failure.json").read_text(encoding="utf-8")
    )
    assert persisted["exception_type"] == "BrokenPipeError"
    assert persisted["partial_model_loading_policy"] == partial_policy
