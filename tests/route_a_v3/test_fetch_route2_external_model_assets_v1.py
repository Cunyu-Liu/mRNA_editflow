from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/fetch_route2_external_model_assets_v1.sh"


def test_external_asset_fetches_use_immutable_revisions() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe" in source
    assert "c575f9cdca0cac1ffa88eb18e4435fdfbc674b08" in source
    assert "7d6e73ad3b48e042b378f9a788a56ccb4d573a27" in source
    assert "/master/" not in source
    assert "/resolve/main/" not in source
