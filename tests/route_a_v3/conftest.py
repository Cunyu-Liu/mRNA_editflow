"""Shared fixtures for Route A V3 A0 static bundle tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "route_a_v3" / "validate_a0_bundle.py"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def validator():
    spec = importlib.util.spec_from_file_location("route_a_v3_validate_a0_bundle", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def bundle_documents(validator, repo_root):
    return validator.load_bundle_documents(repo_root)
