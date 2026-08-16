from __future__ import annotations

from types import SimpleNamespace

import pytest

from core import route2_gpu_failure_evidence as gpu_evidence


def _mock_cuda(monkeypatch: pytest.MonkeyPatch, *, torch_uuid: str, physical_uuid: str) -> None:
    monkeypatch.setattr(gpu_evidence.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_evidence.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(
        gpu_evidence.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(name="MIG", uuid=torch_uuid),
    )
    monkeypatch.setattr(gpu_evidence.torch.cuda, "mem_get_info", lambda _index: (100, 200))
    monkeypatch.setattr(gpu_evidence, "_physical_gpu_uuid", lambda _index: physical_uuid)


def test_declared_physical_index_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cuda(monkeypatch, torch_uuid="PARENT-A", physical_uuid="PARENT-B")
    with pytest.raises(RuntimeError, match="does not belong to physical GPU 7"):
        gpu_evidence.cuda_device_observation(7, require_physical_index_match=True)


def test_matching_parent_and_physical_gpu_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cuda(monkeypatch, torch_uuid="PARENT-A", physical_uuid="PARENT-A")
    observed = gpu_evidence.cuda_device_observation(2, require_physical_index_match=True)
    assert observed["cuda_parent_uuid_matches_declared_physical_index"] is True
    assert observed["declared_physical_gpu_uuid"] == "PARENT-A"
