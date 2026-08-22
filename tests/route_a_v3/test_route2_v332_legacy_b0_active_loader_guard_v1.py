import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
B0_SCRIPT_DIR = ROOT / "d1_staging/scripts/b0"
sys.path.insert(0, str(B0_SCRIPT_DIR))

from legacy_split_guard import (  # noqa: E402
    LEGACY_B0_SPLITS_DIR,
    LegacyB0SplitLoadError,
    reject_legacy_b0_splits,
)


LEGACY_SIZES = {
    "split_study_disjoint.jsonl": 11462850,
    "split_cross_region_transfer.jsonl": 11299013,
    "split_5utr_source_disjoint.jsonl": 7638905,
    "split_3utr_source_disjoint.jsonl": 4338809,
}

ENTRYPOINTS = (
    (
        "d1_staging/scripts/b0/audit_split_manifests.py",
        "--splits-dir",
    ),
    (
        "d1_staging/scripts/b0/eval_tracks.py",
        "--splits-dir",
    ),
    (
        "d1_staging/scripts/b0/leakage_audit.py",
        "--splits-dir",
    ),
    (
        "d1_staging/scripts/fm0/fm0_exposure_audit.py",
        "--b0-splits-dir",
    ),
)


def test_guard_rejects_only_the_superseded_repository_root(tmp_path: Path) -> None:
    assert LEGACY_B0_SPLITS_DIR == (ROOT / "data/b0_splits").resolve()
    with pytest.raises(LegacyB0SplitLoadError, match="SUPERSEDED_NOT_LOADABLE"):
        reject_legacy_b0_splits(ROOT / "data/b0_splits")
    allowed = tmp_path / "versioned_split_registry"
    assert reject_legacy_b0_splits(allowed) == allowed.resolve()


@pytest.mark.parametrize(("script", "split_option"), ENTRYPOINTS)
def test_each_legacy_entrypoint_fails_before_reading_inputs(
    script: str, split_option: str
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / script),
            split_option,
            str(ROOT / "data/b0_splits"),
            "--canonical-records",
            str(ROOT / "does-not-exist-for-negative-loader-test.jsonl"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "SUPERSEDED_NOT_LOADABLE" in output
    assert "does-not-exist-for-negative-loader-test" not in output


def test_guarded_entrypoints_do_not_require_payloads_in_current_head() -> None:
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            *(f"data/b0_splits/{name}" for name in LEGACY_SIZES),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert tracked == []


def test_all_four_reader_sources_call_the_guard_before_loading() -> None:
    expected_markers = {
        "d1_staging/scripts/b0/audit_split_manifests.py": (
            "audit_all_splits",
            "splits_dir = reject_legacy_b0_splits(splits_dir)",
            'print("Loading canonical records for sequence lookup...")',
        ),
        "d1_staging/scripts/b0/eval_tracks.py": (
            "run_eval_track_audit",
            "splits_dir = str(reject_legacy_b0_splits(splits_dir))",
            'print("Loading canonical records...")',
        ),
        "d1_staging/scripts/b0/leakage_audit.py": (
            "run_leakage_audit",
            "splits_dir = reject_legacy_b0_splits(splits_dir)",
            'print("Loading canonical records...")',
        ),
        "d1_staging/scripts/fm0/fm0_exposure_audit.py": (
            "run_exposure_audit",
            "b0_splits_dir = reject_legacy_b0_splits(b0_splits_dir)",
            "cfg = load_config()",
        ),
    }
    for path, (function_name, guard_marker, next_marker) in expected_markers.items():
        text = (ROOT / path).read_text(encoding="utf-8")
        function_text = text.split(f"def {function_name}", 1)[1].split("\ndef ", 1)[0]
        assert guard_marker in function_text
        assert function_text.index(guard_marker) < function_text.index(next_marker)
