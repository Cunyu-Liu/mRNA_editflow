from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.route_a_v3.adjudicate_route2_xeditsetflow_v4_screen import (
    _write_atomic_terminal,
)


def test_setflow_screen_gate_is_atomically_published_and_stale_partial_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "screen_gate.json"
    payload = {"status": "XEDITSETFLOW_V4_SCREEN_NO_GO", "passed": False}
    _write_atomic_terminal(output, payload)
    partial = output.with_suffix(output.suffix + ".partial")
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not partial.exists()

    stale_output = tmp_path / "second_gate.json"
    stale_partial = stale_output.with_suffix(stale_output.suffix + ".partial")
    stale_partial.write_text("interrupted", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial SetFlow V4 screen gate"):
        _write_atomic_terminal(stale_output, payload)
    assert not stale_output.exists()
    assert stale_partial.read_text(encoding="utf-8") == "interrupted"
